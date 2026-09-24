"""Strict functional-equivalence rules used by the V7 recovery.

The rules deliberately stay narrower than semantic similarity:
A = JSON object key order, B = JSON number spelling under a numeric schema,
C = an explicit runtime default or a schema-declared unordered collection.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter
from fc_class_kd.verifier.parser import serialize_function_call
from fc_class_kd.verifier.schema_checker import find_tool_schema, normalize_param_type


@dataclass(frozen=True)
class EquivalentVariant:
    category: str
    call: FunctionCall
    rule: str


def _reordered_dict(value: dict[str, Any]) -> dict[str, Any] | None:
    if len(value) < 2:
        return None
    keys = list(value)
    new_keys = list(reversed(keys))
    if new_keys == keys:
        new_keys = sorted(keys, reverse=keys == sorted(keys))
    if new_keys == keys:
        return None
    return {key: copy.deepcopy(value[key]) for key in new_keys}


def _replace_first_nested_dict(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = _reordered_dict(value)
        if changed is not None:
            return changed, True
        for key in value:
            child, did_change = _replace_first_nested_dict(value[key])
            if did_change:
                result = copy.deepcopy(value)
                result[key] = child
                return result, True
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child, did_change = _replace_first_nested_dict(item)
            if did_change:
                result = copy.deepcopy(value)
                result[index] = child
                return result, True
    return copy.deepcopy(value), False


def _runtime_defaults(record: FCRecord) -> dict[str, Any]:
    """Return only defaults explicitly accepted by the recovered data manifest."""
    accepted = dict(record.metadata.get("v7_safe_defaults", {}))
    tool = find_tool_schema(record.tools, record.gold_call.name)
    if tool is None:
        return accepted
    for key, spec in tool.parameters.items():
        if not spec.required and spec.default is not None:
            accepted.setdefault(key, copy.deepcopy(spec.default))
    return accepted


def _build_runtime_variant(record: FCRecord) -> EquivalentVariant | None:
    tool = find_tool_schema(record.tools, record.gold_call.name)
    if tool is None:
        return None
    defaults = _runtime_defaults(record)
    for key in record.gold_call.arguments:
        if key not in defaults:
            continue
        value = record.gold_call.arguments[key]
        default = defaults[key]
        if type(value) is type(default) and value == default:
            args = copy.deepcopy(record.gold_call.arguments)
            del args[key]
            return EquivalentVariant(
                "C",
                FunctionCall(name=record.gold_call.name, arguments=args),
                f"explicit_runtime_default:{key}",
            )
    for key, value in record.gold_call.arguments.items():
        spec = tool.parameters.get(key)
        if spec is None or not spec.unordered or not isinstance(value, list) or len(value) < 2:
            continue
        args = copy.deepcopy(record.gold_call.arguments)
        args[key] = list(reversed(value))
        if args[key] != value:
            return EquivalentVariant(
                "C",
                FunctionCall(name=record.gold_call.name, arguments=args),
                f"schema_unordered_collection:{key}",
            )
    return None


def _numeric_paths(value: Any, allow_nested: bool) -> Iterator[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield (), value
        return
    if not allow_nested:
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            for suffix, numeric in _numeric_paths(child, True):
                yield (index, *suffix), numeric
    elif isinstance(value, dict):
        for key, child in value.items():
            for suffix, numeric in _numeric_paths(child, True):
                yield (key, *suffix), numeric


def _replace_path(value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    result = copy.deepcopy(value)
    cursor = result
    for item in path[:-1]:
        cursor = cursor[item]
    cursor[path[-1]] = replacement
    return result


def _build_numeric_variant(record: FCRecord) -> EquivalentVariant | None:
    tool = find_tool_schema(record.tools, record.gold_call.name)
    if tool is None:
        return None
    for key, value in record.gold_call.arguments.items():
        spec = tool.parameters.get(key)
        if spec is None:
            continue
        kind = normalize_param_type(spec.type)
        raw = spec.type.lower()
        scalar_number = kind == "number"
        nested_number = kind == "array" and ("float" in raw or "number" in raw)
        if not scalar_number and not nested_number:
            continue
        for path, numeric in _numeric_paths(value, nested_number):
            if isinstance(numeric, int):
                replacement: int | float = float(numeric)
            elif numeric.is_integer():
                replacement = int(numeric)
            else:
                continue
            args = copy.deepcopy(record.gold_call.arguments)
            args[key] = _replace_path(value, path, replacement)
            return EquivalentVariant(
                "B",
                FunctionCall(name=record.gold_call.name, arguments=args),
                f"json_number_spelling:{key}",
            )
    return None


def _build_object_variant(record: FCRecord) -> EquivalentVariant | None:
    args, changed = _replace_first_nested_dict(record.gold_call.arguments)
    if not changed:
        return None
    return EquivalentVariant(
        "A",
        FunctionCall(name=record.gold_call.name, arguments=args),
        "json_object_key_order",
    )


def _canonicalize_value(value: Any, spec: ToolParameter | None) -> Any:
    kind = normalize_param_type(spec.type) if spec is not None else None
    raw = spec.type.lower() if spec is not None else ""
    if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _canonicalize_value(value[key], None) for key in sorted(value)}
    if isinstance(value, list):
        nested_numeric = kind == "array" and ("float" in raw or "number" in raw)
        normalized = [
            float(item)
            if nested_numeric and isinstance(item, (int, float)) and not isinstance(item, bool)
            else _canonicalize_value(item, None)
            for item in value
        ]
        if spec is not None and spec.unordered:
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        return normalized
    return value


def canonical_call(record: FCRecord, call: FunctionCall) -> FunctionCall:
    tool = find_tool_schema(record.tools, call.name)
    if tool is None:
        args = {
            key: _canonicalize_value(call.arguments[key], None) for key in sorted(call.arguments)
        }
        return FunctionCall(name=call.name, arguments=args)
    args = copy.deepcopy(call.arguments)
    for key, default in _runtime_defaults(record).items():
        args.setdefault(key, copy.deepcopy(default))
    normalized = {
        key: _canonicalize_value(args[key], tool.parameters.get(key)) for key in sorted(args)
    }
    return FunctionCall(name=call.name, arguments=normalized)


def canonical_signature(record: FCRecord, call: FunctionCall) -> str:
    canonical = canonical_call(record, call)
    return json.dumps(
        canonical.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_strict_equivalent(record: FCRecord) -> EquivalentVariant | None:
    """Build one non-trivial equivalent with fixed C→B→A precedence."""
    gold_surface = serialize_function_call(record.gold_call, sort_keys=False)
    gold_signature = canonical_signature(record, record.gold_call)
    for builder in (_build_runtime_variant, _build_numeric_variant, _build_object_variant):
        variant = builder(record)
        if variant is None:
            continue
        surface = serialize_function_call(variant.call, sort_keys=False)
        if surface == gold_surface:
            continue
        if canonical_signature(record, variant.call) != gold_signature:
            raise ValueError(f"{record.id}: equivalence rule {variant.rule} changed semantics")
        return variant
    return None
