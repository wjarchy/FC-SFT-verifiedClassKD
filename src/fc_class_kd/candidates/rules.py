"""Rule-based counterfactual candidates (no model required)."""

from __future__ import annotations

import copy
import random
from typing import List, Optional

from fc_class_kd.types import FCRecord, FunctionCall


def _key_reorder_call(call: FunctionCall) -> Optional[FunctionCall]:
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


def _wrong_function_call(record: FCRecord, rng: random.Random) -> Optional[FunctionCall]:
    gold_name = record.gold_call.name
    alts = [t.name for t in record.tools if t.name != gold_name]
    if not alts:
        return None
    name = rng.choice(alts)
    tool = next(t for t in record.tools if t.name == name)
    args = {}
    for k, spec in tool.parameters.items():
        if spec.default is not None:
            args[k] = copy.deepcopy(spec.default)
        elif spec.required:
            if spec.enum:
                args[k] = spec.enum[0]
            elif spec.type in ("int", "integer"):
                args[k] = 1
            elif spec.type in ("float", "number"):
                args[k] = 1.0
            elif spec.type in ("bool", "boolean"):
                args[k] = True
            else:
                args[k] = "placeholder"
    return FunctionCall(name=name, arguments=args)


def _missing_required_call(record: FCRecord) -> Optional[FunctionCall]:
    tool = next((t for t in record.tools if t.name == record.gold_call.name), None)
    if tool is None:
        return None
    required = [k for k, p in tool.parameters.items() if p.required]
    if not required:
        return None
    args = dict(record.gold_call.arguments)
    del args[required[0]]
    return FunctionCall(name=record.gold_call.name, arguments=args)


def _wrong_key_call(record: FCRecord) -> Optional[FunctionCall]:
    args = dict(record.gold_call.arguments)
    if not args:
        return None
    key = next(iter(args))
    args["__wrong_key__"] = args.pop(key)
    return FunctionCall(name=record.gold_call.name, arguments=args)


def _wrong_type_call(record: FCRecord) -> Optional[FunctionCall]:
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


def _invalid_json_call(record: FCRecord) -> FunctionCall:
    return FunctionCall(name="", arguments={"broken": None})


def build_rule_candidates(
    record: FCRecord,
    seed: int = 42,
    include_invalid: bool = True,
) -> List[FunctionCall]:
    """
    Build partial K=6 pool without student on-policy candidates.
    Returns unique candidates; student_hard_error slot left empty (filled later).
    """
    rng = random.Random(seed)
    candidates: List[FunctionCall] = [record.gold_call]

    strict = _key_reorder_call(record.gold_call)
    if strict and strict.arguments != record.gold_call.arguments:
        candidates.append(strict)

    wrong_fn = _wrong_function_call(record, rng)
    if wrong_fn:
        candidates.append(wrong_fn)

    missing = _missing_required_call(record)
    if missing:
        candidates.append(missing)

    wrong_key = _wrong_key_call(record)
    if wrong_key:
        candidates.append(wrong_key)

    wrong_type = _wrong_type_call(record)
    if wrong_type:
        candidates.append(wrong_type)

    if include_invalid:
        candidates.append(_invalid_json_call(record))

    # dedupe by serialized form
    seen = set()
    unique: List[FunctionCall] = []
    for c in candidates:
        key = (c.name, tuple(sorted(c.arguments.items(), key=lambda x: x[0])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique
