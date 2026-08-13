"""Rule-based counterfactual candidates (no model required)."""

from __future__ import annotations

import copy
import random

from fc_class_kd.types import CandidateRecord, FCRecord, FunctionCall
from fc_class_kd.verifier.parser import serialize_function_call
from fc_class_kd.verifier.schema_checker import normalize_param_type


def _representative_value(raw_type: str) -> object:
    first = raw_type.split("|", 1)[0]
    kind = normalize_param_type(first)
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    if kind == "null":
        return None
    return "placeholder"


def _key_reorder_call(call: FunctionCall) -> FunctionCall | None:
    if len(call.arguments) < 2:
        return None
    keys = list(call.arguments.keys())
    if keys == sorted(keys):
        keys.reverse()
    else:
        keys.sort()
    return FunctionCall(
        name=call.name,
        arguments={k: call.arguments[k] for k in keys},
    )


def _wrong_function_call(record: FCRecord, rng: random.Random) -> FunctionCall | None:
    gold_name = record.gold_call.name
    alts = [t.name for t in record.tools if t.name != gold_name]
    if not alts:
        return None
    name = rng.choice(alts)
    tool = next(t for t in record.tools if t.name == name)
    args = {}
    for k, spec in tool.parameters.items():
        if spec.required:
            if spec.enum:
                args[k] = copy.deepcopy(spec.enum[0])
            else:
                args[k] = _representative_value(spec.type)
    return FunctionCall(name=name, arguments=args)


def _missing_required_call(record: FCRecord) -> FunctionCall | None:
    tool = next((t for t in record.tools if t.name == record.gold_call.name), None)
    if tool is None:
        return None
    required = [k for k, p in tool.parameters.items() if p.required]
    if not required:
        return None
    args = dict(record.gold_call.arguments)
    del args[required[0]]
    return FunctionCall(name=record.gold_call.name, arguments=args)


def _wrong_key_call(record: FCRecord) -> FunctionCall | None:
    args = dict(record.gold_call.arguments)
    if not args:
        return None
    key = next(iter(args))
    args["__wrong_key__"] = args.pop(key)
    return FunctionCall(name=record.gold_call.name, arguments=args)


def _wrong_type_call(record: FCRecord) -> FunctionCall | None:
    tool = next((t for t in record.tools if t.name == record.gold_call.name), None)
    if tool is None:
        return None
    for key, spec in tool.parameters.items():
        if key not in record.gold_call.arguments:
            continue
        val = record.gold_call.arguments[key]
        t = spec.type.lower()
        if t in ("int", "integer") and isinstance(val, int):
            return FunctionCall(
                name=record.gold_call.name,
                arguments={**record.gold_call.arguments, key: str(val)},
            )
        if t in ("bool", "boolean") and isinstance(val, bool):
            return FunctionCall(
                name=record.gold_call.name,
                arguments={**record.gold_call.arguments, key: "not_bool"},
            )
    return None


def build_rule_candidates(
    record: FCRecord,
    seed: int = 42,
    include_invalid: bool = True,
) -> list[CandidateRecord]:
    """
    Build partial K=6 pool without student on-policy candidates.
    Returns unique candidates; student_hard_error slot left empty (filled later).
    """
    rng = random.Random(seed)
    candidates: list[CandidateRecord] = [
        CandidateRecord(
            sample_id=record.id,
            source="gold",
            serialized_call=serialize_function_call(record.gold_call, sort_keys=False),
            parsed_call=record.gold_call,
        )
    ]

    strict = _key_reorder_call(record.gold_call)
    if strict:
        strict_surface = serialize_function_call(strict, sort_keys=False)
        if strict_surface != candidates[0].serialized_call:
            candidates.append(
                CandidateRecord(
                    sample_id=record.id,
                    source="strict_equivalent_key_reorder",
                    serialized_call=strict_surface,
                    parsed_call=strict,
                )
            )

    wrong_fn = _wrong_function_call(record, rng)
    if wrong_fn:
        candidates.append(
            CandidateRecord(
                sample_id=record.id,
                source="wrong_function",
                serialized_call=serialize_function_call(wrong_fn, sort_keys=False),
                parsed_call=wrong_fn,
            )
        )

    missing = _missing_required_call(record)
    if missing:
        candidates.append(
            CandidateRecord(
                sample_id=record.id,
                source="missing_required",
                serialized_call=serialize_function_call(missing, sort_keys=False),
                parsed_call=missing,
            )
        )

    wrong_key = _wrong_key_call(record)
    if wrong_key:
        candidates.append(
            CandidateRecord(
                sample_id=record.id,
                source="wrong_key",
                serialized_call=serialize_function_call(wrong_key, sort_keys=False),
                parsed_call=wrong_key,
            )
        )

    wrong_type = _wrong_type_call(record)
    if wrong_type:
        candidates.append(
            CandidateRecord(
                sample_id=record.id,
                source="wrong_type",
                serialized_call=serialize_function_call(wrong_type, sort_keys=False),
                parsed_call=wrong_type,
            )
        )

    if include_invalid:
        candidates.append(
            CandidateRecord(
                sample_id=record.id,
                source="invalid_json",
                serialized_call='{"name":',
                parsed_call=None,
            )
        )

    # Only exact surface duplicates are removed. Canonically equivalent but differently
    # serialized candidates must remain distinct for Candidate-vs-Class KD.
    seen: set[str] = set()
    unique: list[CandidateRecord] = []
    for c in candidates:
        if c.serialized_call in seen:
            continue
        seen.add(c.serialized_call)
        unique.append(c)
    return unique
